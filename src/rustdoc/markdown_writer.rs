export writeinstr;
export writer;
export writer_util;
export make_writer;
export future_writer;

enum writeinstr {
    write(str),
    done
}

type writer = fn~(+writeinstr);

impl writer_util for writer {
    fn write_str(str: str) {
        self(write(str));
    }

    fn write_line(str: str) {
        self.write_str(str + "\n");
    }

    fn write_done() {
        self(done)
    }
}

fn make_writer(config: config::config) -> writer {
    markdown_writer(config)
}

fn markdown_writer(config: config::config) -> writer {
    let filename = make_filename(config);
    let ch = task::spawn_listener {|po: comm::port<writeinstr>|
        let markdown = "";
        let keep_going = true;
        while keep_going {
            alt comm::recv(po) {
              write(s) { markdown += s; }
              done { keep_going = false; }
            }
        }
        write_file(filename, markdown);
    };

    fn~(+instr: writeinstr) {
        comm::send(ch, instr);
    }
}

fn make_filename(config: config::config) -> str {
    import std::fs;
    let cratefile = fs::basename(config.input_crate);
    let cratename = tuple::first(fs::splitext(cratefile));
    fs::connect(config.output_dir, cratename + ".md")
}

fn write_file(path: str, s: str) {
    import std::io;
    import std::io::writer_util;

    alt io::file_writer(path, [io::create, io::truncate]) {
      result::ok(writer) {
        writer.write_str(s);
      }
      result::err(e) { fail e }
    }
}

#[test]
fn should_use_markdown_file_name_based_off_crate() {
    let config = {
        output_dir: "output/dir"
        with config::default_config("input/test.rc")
    };
    assert make_filename(config) == "output/dir/test.md";
}

fn future_writer() -> (writer, future::future<str>) {
    let port = comm::port();
    let chan = comm::chan(port);
    let writer = fn~(+instr: writeinstr) {
        comm::send(chan, copy instr);
    };
    let future = future::from_fn {||
        let res = "";
        while true {
            alt comm::recv(port) {
              write(s) { res += s }
              done { break }
            }
        }
        res
    };
    (writer, future)
}
