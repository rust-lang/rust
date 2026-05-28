// Test `static_mut_refs` lint.

//@ revisions: e2021 e2024

//@ [e2021] edition:2021
//@ [e2021] run-pass

//@ [e2024] edition:2024

static mut FOO: (u32, u32) = (1, 2);

macro_rules! bar {
    ($x:expr) => {
        &mut ($x.0)
        //[e2021]~^ WARN creating a mutable reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR creating a mutable reference to mutable static [static_mut_refs]
    };
}

static mut STATIC: i64 = 1;

fn main() {
    static mut X: i32 = 1;

    static mut Y: i32 = 1;

    struct TheStruct {
        pub value: i32,
    }
    struct MyStruct {
        pub value: i32,
        pub s: TheStruct,
    }

    static mut A: MyStruct = MyStruct { value: 1, s: TheStruct { value: 2 } };

    unsafe {
        let _y = &X;
        //[e2021]~^ WARN shared reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR shared reference to mutable static [static_mut_refs]

        let _y = &mut X;
        //[e2021]~^ WARN mutable reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR mutable reference to mutable static [static_mut_refs]

        let _z = &raw mut X;

        let _p = &raw const X;

        let ref _a = X;
        //[e2021]~^ WARN shared reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR shared reference to mutable static [static_mut_refs]

        let (_b, _c) = (&X, &Y);
        //[e2021]~^ WARN shared reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR shared reference to mutable static [static_mut_refs]
        //[e2021]~^^^ WARN shared reference to mutable static [static_mut_refs]
        //[e2024]~^^^^ ERROR shared reference to mutable static [static_mut_refs]

        foo(&X);
        //[e2021]~^ WARN shared reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR shared reference to mutable static [static_mut_refs]

        static mut Z: &[i32; 3] = &[0, 1, 2];

        let _ = Z.len();
        //[e2021]~^ WARN creating a shared reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR creating a shared reference to mutable static [static_mut_refs]

        let _ = Z[0];

        let _ = format!("{:?}", Z);
        //[e2021]~^ WARN creating a shared reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR creating a shared reference to mutable static [static_mut_refs]

        let _v = &A.value;
        //[e2021]~^ WARN creating a shared reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR creating a shared reference to mutable static [static_mut_refs]

        let _s = &A.s.value;
        //[e2021]~^ WARN creating a shared reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR creating a shared reference to mutable static [static_mut_refs]

        let ref _v = A.value;
        //[e2021]~^ WARN creating a shared reference to mutable static [static_mut_refs]
        //[e2024]~^^ ERROR creating a shared reference to mutable static [static_mut_refs]

        let _x = bar!(FOO);

        STATIC += 1;
    }
}

fn foo<'a>(_x: &'a i32) {}
