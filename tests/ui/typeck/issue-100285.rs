fn foo(n: i32) -> i32 {
    for i in 0..0 {
    //~^ ERROR: mismatched types [E0308]
       if n < 0 {
        return i;
        } else if n < 10 {
          return 1;
        } else if n < 20 {
          return 2;
        } else if n < 30 {
          return 3;
        } else if n < 40 {
          return 4;
        } else {
          return 5;
        }

    }
    //~| help: return a value for the case when the loop has zero elements to iterate on, or consider changing the return type to account for that possibility
}

fn main() {}
