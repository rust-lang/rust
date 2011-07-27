// -*- rust -*-

// error-pattern: attempted dynamic environment-capture

fn f(x: bool) { }

obj foobar(x: bool)
    {drop {
         let y = x;
         fn test() { f(y); }
     }
}

fn main() { }