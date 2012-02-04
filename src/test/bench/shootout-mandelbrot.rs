//  based on:
//  http://shootout.alioth.debian.org/u64q/program.php?test=mandelbrot&lang=python3&id=2
//
//  takes 2 optional numeric args: square image size and yield frequency
//  in the shootout, they use 16000 as image size
//  yield frequency doesn't seem to have much effect
//
//  writes pbm image to stdout

use std;
import std::io::writer_util;

type cmplx = {re: f64, im: f64};
type line = {i: uint, b: [u8]};

impl arith for cmplx {
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
	let z = {re: 0., im: 0.};
	let i = 0;
	let in = true;
	while i < 50 {
		z = z*z + x;
		if cabs(z) >= 4. {
			in = false;
			break;
		}
		i += 1;
	}
	in
}

fn fillbyte(x: cmplx, incr: f64) -> u8 {
	let rv = 0_u8;
	let i = 0_u8;
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
	let crv = [];
	let incr = 2./(size as f64);
	let y = incr*(i as f64) - 1.;
	let xincr = 8.*incr;
	uint::range(0_u, size/8_u) {
		|j|
		let x = {re: xincr*(j as f64) - 1.5, im: y};
		crv += [fillbyte(x, incr)];
	};
	comm::send(ch, {i:i, b:crv});
}

fn writer(writech: comm::chan<comm::chan<line>>, size: uint)
{
	let p: comm::port<line> = comm::port();
	let ch = comm::chan(p);
	comm::send(writech, ch);
	let cout = std::io::stdout();
	cout.write_line("P4");
	cout.write_line(#fmt("%u %u", size, size));
	let lines = std::map::new_uint_hash();
	let done = 0_u;
	let i = 0_u;
	while i < size {
		let aline = comm::recv(p);
		if aline.i == done {
			#debug("W %u", aline.i);
			cout.write(aline.b);
			done += 1_u;
			let prev = done;
			while prev <= i {
				if lines.contains_key(prev) {
					#debug("WS %u", prev);
					cout.write(lines.get(prev));
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

fn main(argv: [str])
{
	let size = if vec::len(argv) < 2_u {
		80u
	}
	else {
		uint::from_str(argv[1])
	};
	let yieldevery = if vec::len(argv) < 3_u {
		10_u
	}
	else {
		uint::from_str(argv[2])
	};
	let writep = comm::port();
	let writech = comm::chan(writep);
	task::spawn {
		|| writer(writech, size);
	};
	let ch = comm::recv(writep);
	uint::range(0_u, size) {
		|j| task::spawn {
			|| chanmb(j, size, ch);
		};
		if j % yieldevery == 0_u {
			#debug("Y %u", j);
			task::yield();
		};
	};
}
