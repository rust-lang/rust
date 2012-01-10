fn main() {
    let a =
        alt 10 { x if x < 7 { 1 } x if x < 11 { 2 } 10 { 3 } _ { 4 } };
    assert (a == 2);

    let b =
        alt {x: 10, y: 20} {
          x if x.x < 5 && x.y < 5 { 1 }
          {x: x, y: y} if x == 10 && y == 20 { 2 }
          {x: x, y: y} { 3 }
        };
    assert (b == 2);
}
