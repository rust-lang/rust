extern mod notstd(name = "std");

use notstd::ebml;
use notstd::json;
use notstd::serialization;
use notstd::net;
use notstd::cell;
use notstd::uv;
use notstd::flatpipe;

pub mod proctask {
    use core::pipes::{Port, Chan};
    use core::path::Path;
    use core::run::spawn_process;
    use core::io::{Reader, Writer};
    use serialization::{Serializable, Deserializable};
    use flatpipe::{FlatChan, FlatPort};
    use flatpipe::serial::{reader_port, writer_chan};

    pub use ProcSerializer = ebml::writer::Serializer;
    pub use ProcDeserializer = ebml::reader::Deserializer;

    pub fn child_proc_stream<CM: Send Serializable<ProcSerializer>, PM: Send Deserializable<ProcDeserializer>>(prog: Path, args: ~[~str]) -> (Chan<CM>, Port<PM>) {

        let (stdin_chan, stdin_port) = pipes::stream();
        let (stdout_chan, stdout_port) = pipes::stream();

        let pipe_in = os::pipe();
        let pipe_out = os::pipe();
        let pipe_err = os::pipe();
        let pid = spawn_process(prog.to_str(), args, &None, &None,
                                pipe_in.in, pipe_out.out, pipe_err.out);

        os::close(pipe_in.in);
        os::close(pipe_out.out);
        os::close(pipe_err.out);

        if pid == -1i32 {
            os::close(pipe_in.out);
            os::close(pipe_out.in);
            os::close(pipe_err.in);
            fail;
        }

        do task::spawn_sched(task::SingleThreaded) |move stdin_port| {
            let writer = io::fd_writer(pipe_in.out, false);
            adapt_port_to_writer(&stdin_port, move writer);
            // XXX: What if above fails?
            os::close(pipe_in.out);
        }

        do task::spawn_sched(task::SingleThreaded) |move stdout_chan| {
            let file = os::fdopen(pipe_out.in);
            let reader = io::FILE_reader(file, false);
            adapt_reader_to_chan(move reader, &stdout_chan);
            // XXX: What if above fails?
            os::close(pipe_out.in);
            os::close(pipe_err.in);
        }

        return (move stdin_chan, move stdout_port);
    }

    pub fn parent_proc_stream<CM: Send Serializable<ProcSerializer>, PM: Send Deserializable<ProcDeserializer>>() -> (Chan<CM>, Port<PM>) {
        let (stdin_chan, stdin_port) = pipes::stream();
        let (stdout_chan, stdout_port) = pipes::stream();

        // Copied from core::io. We want to close the streams when we're done with them
        #[nolink]
        extern mod rustrt {
            fn rust_get_stdin() -> *libc::FILE;
        }

        pub fn stdin() -> @Reader { io::FILE_reader(rustrt::rust_get_stdin(), true) }
        pub fn stdout() -> @Writer { io::fd_writer(libc::STDOUT_FILENO as libc::c_int, true) }

        do task::spawn_sched(task::SingleThreaded) |move stdout_port| {
            let writer = io::stdout();
            adapt_port_to_writer(&stdout_port, move writer);
        }

        do task::spawn_sched(task::SingleThreaded) |move stdin_chan| {
            let reader = io::stdin();
            adapt_reader_to_chan(move reader, &stdin_chan);
        }

        return (move stdout_chan, move stdin_port);
    }

    fn adapt_port_to_writer<M: Send Serializable<ProcSerializer>>(port: &Port<M>, writer: @Writer) {
        let flatchan = writer_chan(move writer);
        loop {
            match port.try_recv() {
                Some(move val) => flatchan.send(move val),
                None => break
            }
        }
    }

    fn adapt_reader_to_chan<M: Send Deserializable<ProcDeserializer>>(reader: @Reader, chan: &Chan<M>) {
        let flatport = reader_port(move reader);
        loop {
            match flatport.try_recv() {
                Some(move val) => chan.send(move val),
                None => break
            }
        }
    }

    // XXX
    mod std {
        pub use notstd::serialization;
    }

    #[auto_serialize]
    #[auto_deserialize]
    enum InputMsg {
        Ping(int),
        Exit
    }

    #[auto_serialize]
    #[auto_deserialize]
    enum OutputMsg {
        Pong(int),
        Done
    }

    //#[test]
    pub fn test_ipc(args: &[~str]) {
        let prog = copy args[0];
        if !args.contains(&~"--ipc") {
            let (chan, port): (Chan<InputMsg>, Port<OutputMsg>) = child_proc_stream(Path(prog), ~[~"--ipc"]);
            for int::range(0, 10) |i| {
                debug!("Ping %?", i);
                chan.send(Ping(i));
                match port.recv() {
                    Pong(j) => {
                        debug!("Pong %?", j);
                        assert i == j;
                    }
                    _ => fail
                }
            }
            chan.send(Exit);
            match port.recv() {
                Done => (),
                _ => fail
            }
        } else {
            let (chan, port): (Chan<OutputMsg>, Port<InputMsg>) = parent_proc_stream();
            loop {
                match port.recv() {
                    Ping(i) => chan.send(Pong(i)),
                    Exit => {
                        chan.send(Done);
                        break;
                    }
                }
            }
        }
    }

}

fn main() {
    proctask::test_ipc(os::args());
}