//@ edition:2021..

#![feature(rustc_attrs, stmt_expr_attributes)]

fn main() {
    // Test that we do not try to capture references to fields of packed structs.

    {
        union NonCopyUnion<'a> {
            field: &'a &'a u32,
        }

        #[repr(packed)]
        struct Packed<'a>(NonCopyUnion<'a>);

        let packed = Packed(NonCopyUnion { field: &&42 });
        let _closure = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let _i = unsafe { *packed.0.field };
            //~^ NOTE Capturing packed[(0, 0),(0, 0),Deref] -> ByCopy
            //~| NOTE Min Capture packed[] -> Immutable
        };
    }

    // Test that implicit reborrows don't lead us to capture more precisely than desired

    {
        let m_s: &mut &u32 = &mut &17;
        let _closure = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let _local: &mut _ = m_s;
            //~^ NOTE Capturing m_s[Deref] -> Mutable
            //~| NOTE Min Capture m_s[Deref] -> Mutable
        };
    }

    {
        let m_s: &mut &u32 = &mut &17;
        let _closure = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let _local: &_ = m_s;
            //~^ NOTE Capturing m_s[Deref] -> Immutable
            //~| NOTE Min Capture m_s[Deref] -> Immutable
        };
    }

     {
        let m_s: &&u32 = &&17;
        let _closure = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let _local: &_ = m_s;
            //~^ NOTE Capturing m_s[Deref] -> Immutable
            //~| NOTE Min Capture m_s[Deref] -> Immutable
        };
    }

    // Box

    {
        let capture = Box::new(false);
        let _closure = #[rustc_capture_analysis]  || *capture;
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        //~| NOTE Capturing capture[Deref] -> Immutable
        //~| NOTE Min Capture capture[Deref] -> Immutable
    }

    {
        struct NotCopy;
        let capture = Box::new(NotCopy);
        let _closure = #[rustc_capture_analysis]  || *capture;
        //~^ ERROR First Pass analysis includes:
        //~| ERROR Min Capture analysis includes:
        //~| NOTE Capturing capture[Deref] -> ByValue
        //~| NOTE Min Capture capture[] -> ByValue
    }

    // Examples from RFC

    {
        let x = [0; 1024];
        let c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let y = x;
            //~^ NOTE Capturing x[] -> ByCopy
            //~| NOTE Min Capture x[] -> ByCopy
        };
    }

    {
        let x = &([0; 1024],);
        let c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let y = x.0;
            //~^ NOTE Capturing x[Deref,(0, 0)] -> Immutable
            //~| NOTE Min Capture x[Deref] -> Immutable
        };
    }

    {
        let x = &(&false,);
        let c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let y = x.0;
            //~^ NOTE Capturing x[Deref,(0, 0)] -> ByCopy
            //~| NOTE Min Capture x[Deref,(0, 0)] -> ByCopy
        };
    }

    {
        let s = String::from("S");
        let t = (s, String::from("T"));
        let mut u = (t, String::from("U"));

        let c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            println!("{:?}", u);
            //~^ NOTE Capturing u[] -> Immutable
            //~| NOTE Min Capture u[] -> ByValue
            //~| NOTE u[] used here
            u.1.truncate(0);
            //~^ NOTE Capturing u[(1, 0)] -> Mutable
            drop(u.0.0);
            //~^ NOTE Capturing u[(0, 0),(0, 0)] -> ByValue
            //~| NOTE u[] captured as ByValue here
        };
        c();
    }

    {
        let s = 'S';
        let t = (s, 'T');
        let mut u = (t, 'U');
        let c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            println!("{:?}", u);
            //~^ NOTE Capturing u[] -> Immutable
            //~| NOTE Min Capture u[] -> Mutable
            //~| NOTE u[] used here
            u.1 = '\0';
            //~^ NOTE Capturing u[(1, 0)] -> Mutable
            //~| NOTE u[] captured as Mutable here
            drop(u.0.0); // `u.0.0` captured by ByCopy
            //~^ NOTE Capturing u[(0, 0),(0, 0)] -> ByCopy
        };
        c();
    }

    {
        struct Int(i32);
        struct B<'a>(&'a i32);

        struct MyStruct<'a> {
            a: &'static Int,
            b: B<'a>,
        }

        fn foo<'a, 'b>(m: &'a MyStruct<'b>) -> impl FnMut() + 'static {
            let c = #[rustc_capture_analysis] || drop(&m.a.0);
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            //~| NOTE Capturing m[Deref,(0, 0),Deref,(0, 0)] -> Immutable
            //~| NOTE Min Capture m[Deref,(0, 0),Deref] -> Immutable
            c
        }
    }

    {
        struct MyStruct<'b> {
            a: &'static u32,
            b: &'b u32,
        }

        fn foo<'a, 'b>(m: &'a MyStruct<'b>) -> impl FnMut() + 'static {
            let c = #[rustc_capture_analysis] || drop(m.a);
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            //~| NOTE Capturing m[Deref,(0, 0)] -> ByCopy
            //~| NOTE Min Capture m[Deref,(0, 0)] -> ByCopy
            c
        }
    }

    {
        struct T(String, String);

        let t = T(String::from("foo"), String::from("bar"));
        let t_ptr = &t as *const T;

        let c = #[rustc_capture_analysis] || unsafe {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            println!("{}", (*t_ptr).0);
            //~^ NOTE Capturing t_ptr[Deref,(0, 0)] -> Immutable
            //~| NOTE Min Capture t_ptr[] -> ByCopy
        };
        c();
    }

    {
        union U {
            a: (i32, i32),
            b: bool,
        }
        let u = U { a: (123, 456) };

        let c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let x = unsafe { u.a.0 };
            //~^ NOTE Capturing u[(0, 0),(0, 0)] -> ByCopy
            //~| NOTE Min Capture u[] -> Immutable
        };
        c();

        // This also includes writing to fields.
        let mut u = U { a: (123, 456) };

        let mut c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            u.b = true;
            //~^ NOTE Capturing u[(1, 0)] -> Mutable
            //~| NOTE Min Capture u[] -> Mutable
        };
        c();
    }

    {
        #[derive(Clone, Copy)]
        union U<'a> {
            a: (&'a String, i32),
            b: bool,
        }
        let string: String = "hello world!".to_owned();
        let u = U { a: (&string, 42) };

        let c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let x = unsafe { u.a.0.len() };
            //~^ NOTE Capturing u[(0, 0),(0, 0),Deref] -> Immutable
            //~| NOTE Min Capture u[] -> ByCopy
        };
        c();
    }

    {
        #[repr(packed)]
        struct T(i32, i32);

        let t = T(2, 5);
        let c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let a = t.0;
            //~^ NOTE Capturing t[(0, 0)] -> ByCopy
            //~| NOTE Min Capture t[(0, 0)] -> ByCopy
        };
        // Copies out of `t` are ok.
        let (a, b) = (t.0, t.1);
        c();
    }

    {
        #[repr(packed)]
        struct T(&'static i32, i32);

        let t = T(&2, 5);
        let c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let a = *t.0;
            //~^ NOTE Capturing t[(0, 0),Deref] -> Immutable
            //~| NOTE Min Capture t[(0, 0),Deref] -> Immutable
        };
        // References out of `t` are ok.
        let (a, b) = (t.0, t.1);
        c();
    }

    {
        struct T(String, String);

        let mut t = T(String::new(), String::new());
        let c = #[rustc_capture_analysis] || {
            //~^ ERROR First Pass analysis includes:
            //~| ERROR Min Capture analysis includes:
            let a = &raw const t.1;
            //~^ NOTE Capturing t[(1, 0)] -> Immutable
            //~| NOTE Min Capture t[(1, 0)] -> Immutable
        };
        // The move here is allowed.
        let a = t.0;
        c();
    }
}
