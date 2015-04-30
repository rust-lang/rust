#![feature(plugin)]
#![plugin(clippy)]

const THREE_BITS : i64 = 7;
const EVEN_MORE_REDIRECTION : i64 = THREE_BITS;

#[deny(bad_bit_mask)]
fn main() {
	let x = 5;
	x & 1 == 1; //ok, distinguishes bit 0
	x & 2 == 1; //~ERROR
	x | 1 == 3; //ok, equals x == 2 || x == 3
	x | 3 == 3; //ok, equals x <= 3
	x | 3 == 2; //~ERROR
	
	x & 1 > 1; //~ERROR
	x & 2 > 1; // ok, distinguishes x & 2 == 2 from x & 2 == 0
	x & 2 < 1; // ok, distinguishes x & 2 == 2 from x & 2 == 0
	x | 1 > 1; // ok (if a bit silly), equals x > 1
	x | 2 > 1; //~ERROR
	x | 2 <= 2; // ok (if a bit silly), equals x <= 2
	
	// this also now works with constants
	x & THREE_BITS == 8; //~ERROR
	x | EVEN_MORE_REDIRECTION < 7; //~ERROR
}
