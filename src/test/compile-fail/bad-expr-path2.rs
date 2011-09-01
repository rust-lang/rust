// error-pattern: unresolved name: a

mod m1 {
    mod a { }
}

fn main(args: [istr]) { log m1::a; }
