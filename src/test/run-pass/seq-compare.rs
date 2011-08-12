

fn main() {
    assert ("hello" < "hellr");
    assert ("hello " > "hello");
    assert ("hello" != "there");
    assert (~[1, 2, 3, 4] > ~[1, 2, 3]);
    assert (~[1, 2, 3] < ~[1, 2, 3, 4]);
    assert (~[1, 2, 4, 4] > ~[1, 2, 3, 4]);
    assert (~[1, 2, 3, 4] < ~[1, 2, 4, 4]);
    assert (~[1, 2, 3] <= ~[1, 2, 3]);
    assert (~[1, 2, 3] <= ~[1, 2, 3, 3]);
    assert (~[1, 2, 3, 4] > ~[1, 2, 3]);
    assert (~[1, 2, 3] == ~[1, 2, 3]);
    assert (~[1, 2, 3] != ~[1, 1, 3]);
}