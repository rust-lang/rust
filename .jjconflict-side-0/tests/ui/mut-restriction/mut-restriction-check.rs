//@ aux-build: external-mut-restriction.rs
//@ edition: 2018..
#![feature(mut_restriction)]

extern crate external_mut_restriction as external;

fn change_extern_structs(foo: &mut external::TopLevelS, bar: &mut external::inner::InnerS) {
    foo.x = 1; //~ ERROR field `x` cannot be mutated outside `external`
    foo.y = 1; // ok
    bar.x = 1; //~ ERROR field `x` cannot be mutated outside `external`
    bar.y = 1; // ok
}

fn change_extern_enums(foo: &mut external::TopLevelE, bar: &mut external::inner::InnerE) {
    match foo {
        external::TopLevelE::Var {
            x, //~ ERROR field `x` cannot be mutated outside `external`
            y, // ok
        } => {}
        external::TopLevelE::Tup(
            x, //~ ERROR field `0` cannot be mutated outside `external`
            y, // ok
        ) => {}
    }
    match bar {
        external::inner::InnerE::Var {
            x, //~ ERROR field `x` cannot be mutated outside `external`
            y, // ok
        } => {}
        external::inner::InnerE::Tup(
            x, //~ ERROR field `0` cannot be mutated outside `external`
            y, // ok
        ) => {}
    }
}

fn change_extern_unions(foo: &mut external::TopLevelU, bar: &mut external::inner::InnerU) {
    unsafe {
        foo.x = 1; //~ ERROR field `x` cannot be mutated outside `external`
        foo.y = 1; // ok
        bar.x = 1; //~ ERROR field `x` cannot be mutated outside `external`
        bar.y = 1; // ok
    }
}

pub mod foo {
    pub mod bar {
        #[derive(Default, Clone)]
        pub struct FooS {
            pub mut(self) alpha: u8,
            pub mut(super) beta: u8,
            pub mut(crate) gamma: u8,
        }

        impl FooS {
            pub fn change_inner(&mut self) {
                self.alpha = 1; // ok
                self.beta = 1; // ok
                self.gamma = 1; // ok
            }
        }

        #[derive(Default)]
        pub enum FooE {
            #[default]
            Default,
            Var {
                mut(in crate::foo::bar) alpha: u8,
                mut(in crate::foo) beta: u8,
                gamma: u8,
            },
            Tup(mut(in crate::foo::bar) u8, mut(in crate::foo) u8, u8),
        }

        impl FooE {
            pub fn change_inner(&mut self) {
                match self {
                    FooE::Default => {}
                    FooE::Var { alpha, beta, gamma } => { // ok
                        *alpha = 1;
                        *beta = 1;
                        *gamma = 1;
                    }
                    FooE::Tup(alpha, beta, gamma) => { // ok
                        *alpha = 1;
                        *beta = 1;
                        *gamma = 1;
                    }
                }
            }
        }

        pub union FooU {
            pub mut(in self) alpha: u8,
            pub mut(in super) beta: u8,
            pub mut(in super::super) gamma: u8,
        }

        impl FooU {
            pub fn new() -> Self {
                Self { alpha: 0 }
            }
            pub fn change_inner(&mut self) {
                unsafe {
                    self.alpha = 1; // ok
                    self.beta = 1; // ok
                    self.gamma = 1; // ok
                }
            }
        }

        #[derive(Default)]
        pub struct Bar(
            pub mut(self) u8,
            pub mut(super) u8,
            pub mut(crate) u8,
        );

        pub struct Baz<'a> {
            pub mut(self) array: [u8; 2],
            pub mut(self) vector: Vec<u8>,
            pub mut(self) slice: &'a mut [u8],
        }

        impl<'a> Baz<'a> {
            pub fn new(slice: &'a mut [u8]) -> Self {
                Self { array: [0; 2], vector: vec![0; 2], slice }
            }
        }
    }

    impl bar::FooS {
        pub fn change_outer(&mut self) {
            self.alpha = 1; //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
            self.beta = 1; // ok
            self.gamma = 1; // ok
        }
    }

    impl bar::FooE {
        pub fn change_outer(&mut self) {
            match self {
                bar::FooE::Default => {}
                bar::FooE::Var {
                    alpha, //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
                    beta, // ok
                    gamma, // ok
                } => {
                    *alpha = 1;
                    *beta = 1;
                    *gamma = 1;
                }
                bar::FooE::Tup(
                    alpha, //~ ERROR field `0` cannot be mutated outside `crate::foo::bar`
                    beta, // ok
                    gamma, // ok
                ) => {
                    *alpha = 1;
                    *beta = 1;
                    *gamma = 1;
                }
            }
        }
    }

    impl bar::FooU {
        pub fn change_outer(&mut self) {
            unsafe {
                self.alpha = 1; //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
                self.beta = 1; // ok
                self.gamma = 1; // ok
            }
        }
    }
}

fn change_foos(foo: &mut foo::bar::FooS) {
    foo.alpha = 1; //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
    foo.beta = 1; //~ ERROR field `beta` cannot be mutated outside `crate::foo`
    foo.gamma = 1; // ok
}

// Currently, both borrow check and `mut` restriction error are emitted
fn change_foos_immut(foo: &foo::bar::FooS) {
    foo.alpha = 1; //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
    //~^ ERROR cannot assign to `foo.alpha`, which is behind a `&` reference [E0594]
    foo.beta = 1; //~ ERROR field `beta` cannot be mutated outside `crate::foo`
    //~^ ERROR cannot assign to `foo.beta`, which is behind a `&` reference [E0594]
    foo.gamma = 1; //~ ERROR cannot assign to `foo.gamma`, which is behind a `&` reference [E0594]
}

fn change_fooe(foo: &mut foo::bar::FooE) {
    match foo {
        foo::bar::FooE::Default => {}
        foo::bar::FooE::Var {
            alpha, //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
            beta, //~ ERROR field `beta` cannot be mutated outside `crate::foo`
            gamma, // ok
        } => {
            *alpha = 1;
            *beta = 1;
            *gamma = 1;
        }
        foo::bar::FooE::Tup(
            alpha, //~ ERROR field `0` cannot be mutated outside `crate::foo::bar`
            beta, //~ ERROR field `1` cannot be mutated outside `crate::foo`
            gamma, // ok
        ) => {
            *alpha = 1;
            *beta = 1;
            *gamma = 1;
        }
    }

    // Reborrow as immutable
    match &*foo {
        foo::bar::FooE::Default => {}
        foo::bar::FooE::Var {
            alpha,
            beta,
            gamma,
        } => {
            *alpha = 1; //~ ERROR cannot assign to `*alpha`, which is behind a `&` reference [E0594]
            *beta = 1; //~ ERROR cannot assign to `*beta`, which is behind a `&` reference [E0594]
            *gamma = 1; //~ ERROR cannot assign to `*gamma`, which is behind a `&` reference [E0594]
        }
        foo::bar::FooE::Tup(
            alpha,
            beta,
            gamma,
        ) => {
            *alpha = 1; //~ ERROR cannot assign to `*alpha`, which is behind a `&` reference [E0594]
            *beta = 1; //~ ERROR cannot assign to `*beta`, which is behind a `&` reference [E0594]
            *gamma = 1; //~ ERROR cannot assign to `*gamma`, which is behind a `&` reference [E0594]
        }
    }

    // Another reborrow as immutable
    match &*foo {
        &foo::bar::FooE::Default => {}
        &foo::bar::FooE::Var {
            ref alpha,
            ref beta,
            ref gamma,
        } => {
            *alpha = 1; //~ ERROR cannot assign to `*alpha`, which is behind a `&` reference [E0594]
            *beta = 1; //~ ERROR cannot assign to `*beta`, which is behind a `&` reference [E0594]
            *gamma = 1; //~ ERROR cannot assign to `*gamma`, which is behind a `&` reference [E0594]
        }
        &foo::bar::FooE::Tup(
            ref alpha,
            ref beta,
            ref gamma,
        ) => {
            *alpha = 1; //~ ERROR cannot assign to `*alpha`, which is behind a `&` reference [E0594]
            *beta = 1; //~ ERROR cannot assign to `*beta`, which is behind a `&` reference [E0594]
            *gamma = 1; //~ ERROR cannot assign to `*gamma`, which is behind a `&` reference [E0594]
        }
    }

    // immutbale borrow inside match
    match foo {
        foo::bar::FooE::Default => {}
        foo::bar::FooE::Var {
            ref alpha,
            beta, //~ ERROR field `beta` cannot be mutated outside `crate::foo`
            gamma,
        } => {
            *alpha = 1; //~ ERROR cannot assign to `*alpha`, which is behind a `&` reference [E0594]
            *beta = 1;
            *gamma = 1;
        }
        foo::bar::FooE::Tup(
            ref alpha,
            beta, //~ ERROR field `1` cannot be mutated outside `crate::foo`
            gamma,
        ) => {
            *alpha = 1; //~ ERROR cannot assign to `*alpha`, which is behind a `&` reference [E0594]
            *beta = 1;
            *gamma = 1;
        }
    }

    // use _
    match foo {
        foo::bar::FooE::Default => {}
        foo::bar::FooE::Var {
            alpha: _,
            beta, //~ ERROR field `beta` cannot be mutated outside `crate::foo`
            gamma,
        } => {
            *beta = 1;
            *gamma = 1;
        }
        foo::bar::FooE::Tup(
            _,
            beta, //~ ERROR field `1` cannot be mutated outside `crate::foo`
            gamma,
        ) => {
            *beta = 1;
            *gamma = 1;
        }
    }
}

fn change_fooe_immut(foo: &foo::bar::FooE) {
    match foo {
        foo::bar::FooE::Default => {}
        foo::bar::FooE::Var {
            alpha,
            beta,
            gamma,
        } => {
            *alpha = 1; //~ ERROR cannot assign to `*alpha`, which is behind a `&` reference [E0594]
            *beta = 1; //~ ERROR cannot assign to `*beta`, which is behind a `&` reference [E0594]
            *gamma = 1; //~ ERROR cannot assign to `*gamma`, which is behind a `&` reference [E0594]
        }
        foo::bar::FooE::Tup(
            alpha,
            beta,
            gamma,
        ) => {
            *alpha = 1; //~ ERROR cannot assign to `*alpha`, which is behind a `&` reference [E0594]
            *beta = 1; //~ ERROR cannot assign to `*beta`, which is behind a `&` reference [E0594]
            *gamma = 1; //~ ERROR cannot assign to `*gamma`, which is behind a `&` reference [E0594]
        }
    }
}

fn change_foou(foo: &mut foo::bar::FooU) {
    unsafe {
        foo.alpha = 1; //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
        foo.beta = 1; //~ ERROR field `beta` cannot be mutated outside `crate::foo`
        foo.gamma = 1; // ok
    }
}

fn change_foos_ptr(foo: *mut foo::bar::FooS) {
    // unsafe doesn`t matter
    unsafe {
        (*foo).alpha = 1; //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
        (*foo).beta = 1; //~ ERROR field `beta` cannot be mutated outside `crate::foo`
        (*foo).gamma = 1; // ok
    }
}

fn change_bar(bar: &mut foo::bar::Bar) {
    bar.0 = 1; //~ ERROR field `0` cannot be mutated outside `crate::foo::bar`
    bar.1 = 1; //~ ERROR field `1` cannot be mutated outside `crate::foo`
    bar.2 = 1; // ok
}

fn main() {
    let mut foos = foo::bar::FooS::default();
    foos.alpha = 1; //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
    std::ptr::addr_of_mut!(foos.alpha); //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`

    let _beta = &mut foos.beta; //~ ERROR field `beta` cannot be mutated outside `crate::foo`
    let _gamma = &mut foos.gamma; // ok

    let mut closure = || {
        foos.alpha = 1; //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
        foos.beta = 1; //~ ERROR field `beta` cannot be mutated outside `crate::foo`
        foos.gamma = 1; // ok
    };

    // ok: the mutation occurs inside the function
    closure();
    change_foos(&mut foos);
    change_foos_ptr(&mut foos as *mut _);
    foos.change_inner();
    foos.change_outer();

    // ok: this is the same as turning &T into &mut T, which is unsound
    unsafe { *(&foos.alpha as *const _ as *mut _) = 1; }

    let mut fooe = foo::bar::FooE::default();
    fooe.change_inner();
    fooe.change_outer();
    change_fooe(&mut fooe);

    let mut foou = foo::bar::FooU::new();
    foou.change_inner();
    foou.change_outer();
    change_foou(&mut foou);

    change_bar(&mut foo::bar::Bar::default());

    let mut ls = vec![foo::bar::FooS::default(); 2];
    ls[0].alpha = 1; //~ ERROR field `alpha` cannot be mutated outside `crate::foo::bar`
    ls[0].beta = 1; //~ ERROR field `beta` cannot be mutated outside `crate::foo`
    ls[0].gamma = 1; // ok

    let mut slice = [0; 2];
    let mut baz = foo::bar::Baz::new(&mut slice);
    baz.array[0] = 1; //~ ERROR field `array` cannot be mutated outside `crate::foo::bar`
    baz.vector[0] = 1; //~ ERROR field `vector` cannot be mutated outside `crate::foo::bar`
    baz.slice[0] = 1; //~ ERROR field `slice` cannot be mutated outside `crate::foo::bar`
}
