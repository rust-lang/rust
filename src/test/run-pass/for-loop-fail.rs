fn main() { let x: ~[int] = ~[]; for x.each |_i| { fail ~"moop"; } }
