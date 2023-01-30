// Test for #78438: ensure underline alignment with many tabs on the left, long line on the right

// ignore-tidy-linelength
// ignore-tidy-tab

					fn main() {
						let money = 42i32;
						match money {
							v @ 1 | 2 | 3 => panic!("You gave me too little money {}", v), // Long text here: TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
							//~^ ERROR variable `v` is not bound in all patterns
							v => println!("Enough money {}", v),
						}
					}
