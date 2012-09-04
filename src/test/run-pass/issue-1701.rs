enum pattern { tabby, tortoiseshell, calico }
enum breed { beagle, rottweiler, pug }
type name = ~str;
enum ear_kind { lop, upright }
enum animal { cat(pattern), dog(breed), rabbit(name, ear_kind), tiger }

fn noise(a: animal) -> Option<~str> {
    match a {
      cat(*)    => { Some(~"meow") }
      dog(*)    => { Some(~"woof") }
      rabbit(*) => { None }
      tiger(*)  => { Some(~"roar") }
    }
}

fn main() {
    assert noise(cat(tabby)) == Some(~"meow");
    assert noise(dog(pug)) == Some(~"woof");
    assert noise(rabbit(~"Hilbert", upright)) == None;
    assert noise(tiger) == Some(~"roar");
}