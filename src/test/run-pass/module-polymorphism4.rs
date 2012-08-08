// This isn't really xfailed; it's used by the
// module-polymorphism.rc test
// xfail-test

fn main() {
    let cat1 = cat::inst::meowlycat;
    let cat2 = cat::inst::howlycat;
    let dog = dog::inst::dog;
    assert cat1.says() == ~"cat says 'meow'";
    assert cat2.says() == ~"cat says 'howl'";
    assert dog.says() == ~"dog says 'woof'";
}