// Regression test for #23729

fn main() {
    let fib = {
        struct Recurrence {
            mem: [u64; 2],
            pos: usize,
        }

        impl Iterator for Recurrence {
            //~^ ERROR E0046
            #[inline]
            fn next(&mut self) -> Option<u64> {
                if self.pos < 2 {
                    let next_val = self.mem[self.pos];
                    self.pos += 1;
                    Some(next_val)
                } else {
                    let next_val = self.mem[0] + self.mem[1];
                    self.mem[0] = self.mem[1];
                    self.mem[1] = next_val;
                    Some(next_val)
                }
            }
        }

        Recurrence { mem: [0, 1], pos: 0 }
    };

    for e in fib.take(10) {
        println!("{}", e)
    }
}
