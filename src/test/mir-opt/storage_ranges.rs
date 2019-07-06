fn main() {
    let a = 0;
    {
        let b = &Some(a);
    }
    let c = 1;
}

// END RUST SOURCE
// START rustc.main.nll.0.mir
//     bb0: {
//         StorageLive(_1);
//         _1 = const 0i32;
//         FakeRead(ForLet, _1);
//         StorageLive(_2);
//         StorageLive(_3);
//         StorageLive(_4);
//         StorageLive(_5);
//         _5 = _1;
//         _4 = std::option::Option::<i32>::Some(move _5,);
//         StorageDead(_5);
//         _3 = &_4;
//         FakeRead(ForLet, _3);
//         _2 = ();
//         StorageDead(_4);
//         StorageDead(_3);
//         StorageDead(_2);
//         StorageLive(_6);
//         _6 = const 1i32;
//         FakeRead(ForLet, _6);
//         _0 = ();
//         StorageDead(_6);
//         StorageDead(_1);
//         return;
//      }
// END rustc.main.nll.0.mir
