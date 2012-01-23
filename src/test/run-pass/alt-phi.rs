

enum thing { a, b, c, }

fn foo(it: fn(int)) { it(10); }

fn main() {
    let x = true;
    alt a {
      a { x = true; foo {|_i|} }
      b { x = false; }
      c { x = false; }
    }
}
