fn main() {
    for vec::each(~[{x: 10, y: 20}, {x: 30, y: 0}]) {|elt|
        assert (elt.x + elt.y == 30);
    }
}
