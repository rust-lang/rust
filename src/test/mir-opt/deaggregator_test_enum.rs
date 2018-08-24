enum Baz {
    Empty,
    Foo { x: usize },
}

fn bar(a: usize) -> Baz {
    Baz::Foo { x: a }
}

fn main() {
    let x = bar(10);
    match x {
        Baz::Empty => println!("empty"),
        Baz::Foo { x } => println!("{}", x),
    };
}

// END RUST SOURCE
// START rustc.bar.Deaggregator.before.mir
// bb0: {
//     StorageLive(_2);
//     _2 = _1;
//     _0 = Baz::Foo { x: move _2 };
//     StorageDead(_2);
//     return;
// }
// END rustc.bar.Deaggregator.before.mir
// START rustc.bar.Deaggregator.after.mir
// bb0: {
//     StorageLive(_2);
//     _2 = _1;
//     ((_0 as Foo).0: usize) = move _2;
//     discriminant(_0) = 1;
//     StorageDead(_2);
//     return;
// }
// END rustc.bar.Deaggregator.after.mir
