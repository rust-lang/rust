enum pattern { tabby, tortoiseshell, calico }
enum breed { beagle, rottweiler, pug }
type name = ~str;
enum ear_kind { lop, upright }
enum animal { cat(pattern), dog(breed), rabbit(name, ear_kind), tiger }

fn noise(a: animal) -> option<~str> {
    alt a {
      cat(*)    => { some(~"meow") }
      dog(*)    => { some(~"woof") }
      rabbit(*) => { none }
      tiger(*)  => { some(~"roar") }
    }
}

fn main() {
    assert noise(cat(tabby)) == some(~"meow");
    assert noise(dog(pug)) == some(~"woof");
    assert noise(rabbit(~"Hilbert", upright)) == none;
    assert noise(tiger) == some(~"roar");
}