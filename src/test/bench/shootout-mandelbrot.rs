//  based on:
//  http://shootout.alioth.debian.org/
//   u64q/program.php?test=mandelbrot&lang=python3&id=2
//
//  takes 3 optional args:
//   square image size, defaults to 80_u
//   yield frequency, defaults to 10_u (yield every 10 spawns)
//   output path, default is "" (no output), "-" means stdout
//
//  in the shootout, they use 16000 as image size
//  yield frequency doesn't seem to have much effect
//
//  writes pbm image to output path

use std;
import io::writer_util;
import std::map::hashmap;

type cmplx = {re: f64, im: f64};
type line = {i: uint, b: ~[u8]};

trait times_and_plus {
    fn *(x: cmplx) -> cmplx;
    fn +(x: cmplx) -> cmplx;
}

impl arith of times_and_plus for cmplx {
    fn *(x: cmplx) -> cmplx {
        {re: self.re*x.re - self.im*x.im, im: self.re*x.im + self.im*x.re}
    }

    fn +(x: cmplx) -> cmplx {
        {re: self.re + x.re, im: self.im + x.im}
    }
}

pure fn cabs(x: cmplx) -> f64
{
    x.re*x.re + x.im*x.im
}

fn mb(x: cmplx) -> bool
{
    let mut z = {re: 0f64, im: 0f64};
    let mut i = 0;
    let mut in = true;
    while i < 50 {
        z = z*z + x;
        if cabs(z) >= 4f64 {
            in = false;
            break;
        }
        i += 1;
    }
    in
}

fn fillbyte(x: cmplx, incr: f64) -> u8 {
    let mut rv = 0_u8;
    let mut i = 0_u8;
    while i < 8_u8 {
        let z = {re: x.re + (i as f64)*incr, im: x.im};
        if mb(z) {
            rv += 1_u8 << (7_u8 - i);
        }
        i += 1_u8;
    }
    rv
}

fn chanmb(i: uint, size: uint, ch: comm::chan<line>) -> ()
{
    let mut crv = ~[];
    let incr = 2f64/(size as f64);
    let y = incr*(i as f64) - 1f64;
    let xincr = 8f64*incr;
    for uint::range(0_u, size/8_u) |j| {
        let x = {re: xincr*(j as f64) - 1.5f64, im: y};
        vec::push(crv, fillbyte(x, incr));
    };
    comm::send(ch, {i:i, b:crv});
}

type devnull = {dn: int};

impl of io::writer for devnull {
    fn write(_b: &[const u8]) {}
    fn seek(_i: int, _s: io::seek_style) {}
    fn tell() -> uint {0_u}
    fn flush() -> int {0}
    fn get_type() -> io::writer_type { io::file }
}

fn writer(path: ~str, writech: comm::chan<comm::chan<line>>, size: uint)
{
    let p: comm::port<line> = comm::port();
    let ch = comm::chan(p);
    comm::send(writech, ch);
    let cout: io::writer = alt path {
        ~"" {
            {dn: 0} as io::writer
        }
        ~"-" {
            io::stdout()
        }
        _ {
            result::get(
                io::file_writer(path,
                ~[io::create, io::truncate]))
        }
    };
    cout.write_line(~"P4");
    cout.write_line(#fmt("%u %u", size, size));
    let lines = std::map::uint_hash();
    let mut done = 0_u;
    let mut i = 0_u;
    while i < size {
        let aline = comm::recv(p);
        if aline.i == done {
            #debug("W %u", aline.i);
            cout.write(aline.b);
            done += 1_u;
            let mut prev = done;
            while prev <= i {
                if lines.contains_key(prev) {
                    #debug("WS %u", prev);
                    // FIXME (#2280): this temporary shouldn't be
                    // necessary, but seems to be, for borrowing.
                    let v : ~[u8] = lines.get(prev);
                    cout.write(v);
                    done += 1_u;
                    lines.remove(prev);
                    prev += 1_u;
                }
                else {
                    break
                }
            };
        }
        else {
            #debug("S %u", aline.i);
            lines.insert(aline.i, aline.b);
        };
        i += 1_u;
    }
}

fn main(args: ~[~str]) {
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"4000", ~"10"]
    } else {
        args
    };

    let path = if vec::len(args) < 4_u { ~"" }
    else { args[3] };

    let yieldevery = if vec::len(args) < 3_u { 10_u }
    else { uint::from_str(args[2]).get() };

    let size = if vec::len(args) < 2_u { 80_u }
    else { uint::from_str(args[1]).get() };

    let writep = comm::port();
    let writech = comm::chan(writep);
    do task::spawn {
        writer(path, writech, size);
    };
    let ch = comm::recv(writep);
    for uint::range(0_u, size) |j| {
        task::spawn(|| chanmb(j, size, ch) );
        if j % yieldevery == 0_u {
            #debug("Y %u", j);
            task::yield();
        };
    };
}
