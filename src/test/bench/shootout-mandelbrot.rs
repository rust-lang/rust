// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//  based on:
//  http://shootout.alioth.debian.org/
//   u64q/program.php?test=mandelbrot&lang=python3&id=2
//
//  takes 3 optional args:
//   square image size, defaults to 80_u
//   output path, default is "" (no output), "-" means stdout
//   depth (max iterations per pixel), defaults to 50_u
//
//  in the shootout, they use 16000 as image size, 50 as depth,
//   and write to stdout:
//
//  ./shootout_mandelbrot 16000 "-" 50 > /tmp/mandel.pbm
//
//  writes pbm image to output path

use core::io::WriterUtil;
use core::hashmap::HashMap;

struct cmplx {
    re: f64,
    im: f64
}

impl ops::Mul<cmplx,cmplx> for cmplx {
    fn mul(&self, x: &cmplx) -> cmplx {
        cmplx {
            re: self.re*(*x).re - self.im*(*x).im,
            im: self.re*(*x).im + self.im*(*x).re
        }
    }
}

impl ops::Add<cmplx,cmplx> for cmplx {
    fn add(&self, x: &cmplx) -> cmplx {
        cmplx {
            re: self.re + (*x).re,
            im: self.im + (*x).im
        }
    }
}

struct Line {i: uint, b: ~[u8]}

fn cabs(x: cmplx) -> f64
{
    x.re*x.re + x.im*x.im
}

fn mb(x: cmplx, depth: uint) -> bool
{
    let mut z = x;
    let mut i = 0;
    while i < depth {
        if cabs(z) >= 4_f64 {
            return false;
        }
        z = z*z + x;
        i += 1;
    }
    true
}

fn fillbyte(x: cmplx, incr: f64, depth: uint) -> u8 {
    let mut rv = 0_u8;
    let mut i = 0_u8;
    while i < 8_u8 {
        let z = cmplx {re: x.re + (i as f64)*incr, im: x.im};
        if mb(z, depth) {
            rv += 1_u8 << (7_u8 - i);
        }
        i += 1_u8;
    }
    rv
}

fn chanmb(i: uint, size: uint, depth: uint) -> Line
{
    let bsize = size/8_u;
    let mut crv = vec::with_capacity(bsize);
    let incr = 2_f64/(size as f64);
    let y = incr*(i as f64) - 1_f64;
    let xincr = 8_f64*incr;
    for uint::range(0_u, bsize) |j| {
        let x = cmplx {re: xincr*(j as f64) - 1.5_f64, im: y};
        crv.push(fillbyte(x, incr, depth));
    };
    Line {i:i, b:crv}
}

struct Devnull();

impl io::Writer for Devnull {
    fn write(&self, _b: &const [u8]) {}
    fn seek(&self, _i: int, _s: io::SeekStyle) {}
    fn tell(&self) -> uint {0_u}
    fn flush(&self) -> int {0}
    fn get_type(&self) -> io::WriterType { io::File }
}

fn writer(path: ~str, pport: comm::Port<Line>, size: uint)
{
    let cout: @io::Writer = match path {
        ~"" => {
            @Devnull as @io::Writer
        }
        ~"-" => {
            io::stdout()
        }
        _ => {
            result::get(
                &io::file_writer(&Path(path),
                ~[io::Create, io::Truncate]))
        }
    };
    cout.write_line("P4");
    cout.write_line(fmt!("%u %u", size, size));
    let mut lines: HashMap<uint, Line> = HashMap::new();
    let mut done = 0_u;
    let mut i = 0_u;
    while i < size {
        let aline = pport.recv();
        if aline.i == done {
            debug!("W %u", done);
            cout.write(aline.b);
            done += 1_u;
            let mut prev = done;
            while prev <= i {
                match lines.pop(&prev) {
                    Some(pl) => {
                        debug!("WS %u", prev);
                        cout.write(pl.b);
                        done += 1_u;
                        prev += 1_u;
                    }
                    None => break
                };
            };
        }
        else {
            debug!("S %u", aline.i);
            lines.insert(aline.i, aline);
        };
        i += 1_u;
    }
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"4000", ~"50"]
    } else {
        args
    };

    let depth = if vec::len(args) < 4_u { 50_u }
    else { uint::from_str(args[3]).get() };

    let path = if vec::len(args) < 3_u { ~"" }
    else { copy args[2] };  // FIXME: bad for perf

    let size = if vec::len(args) < 2_u { 80_u }
    else { uint::from_str(args[1]).get() };

    let (pport, pchan) = comm::stream();
    let pchan = comm::SharedChan(pchan);
    for uint::range(0_u, size) |j| {
        let cchan = pchan.clone();
        do task::spawn { cchan.send(chanmb(j, size, depth)) };
    };
    writer(path, pport, size);
}
