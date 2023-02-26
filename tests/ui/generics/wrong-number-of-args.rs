mod no_generics {
    struct Ty;

    type A = Ty;

    type B = Ty<'static>;
    //~^ ERROR struct takes 0 lifetime arguments but 1 lifetime argument
    //~| HELP remove these generics

    type C = Ty<'static, usize>;
    //~^ ERROR struct takes 0 lifetime arguments but 1 lifetime argument
    //~| ERROR struct takes 0 generic arguments but 1 generic argument
    //~| HELP remove this lifetime argument
    //~| HELP remove this generic argument

    type D = Ty<'static, usize, { 0 }>;
    //~^ ERROR struct takes 0 lifetime arguments but 1 lifetime argument
    //~| ERROR struct takes 0 generic arguments but 2 generic arguments
    //~| HELP remove this lifetime argument
    //~| HELP remove these generic arguments
}

mod type_and_type {
    struct Ty<A, B>;

    type A = Ty;
    //~^ ERROR missing generics for struct `type_and_type::Ty`
    //~| HELP add missing

    type B = Ty<usize>;
    //~^ ERROR struct takes 2 generic arguments but 1 generic argument
    //~| HELP add missing

    type C = Ty<usize, String>;

    type D = Ty<usize, String, char>;
    //~^ ERROR struct takes 2 generic arguments but 3 generic arguments
    //~| HELP remove this

    type E = Ty<>;
    //~^ ERROR struct takes 2 generic arguments but 0 generic arguments were supplied
    //~| HELP add missing
}

mod lifetime_and_type {
    struct Ty<'a, T>;

    type A = Ty;
    //~^ ERROR missing generics for struct
    //~| ERROR missing lifetime specifier
    //~| HELP add missing
    //~| HELP consider introducing

    type B = Ty<'static>;
    //~^ ERROR struct takes 1 generic argument but 0 generic arguments
    //~| HELP add missing

    type C = Ty<usize>;
    //~^ ERROR missing lifetime specifier
    //~| HELP consider introducing

    type D = Ty<'static, usize>;

    type E = Ty<>;
    //~^ ERROR struct takes 1 generic argument but 0 generic arguments
    //~| ERROR missing lifetime specifier
    //~| HELP consider introducing
    //~| HELP add missing

    type F = Ty<'static, usize, 'static, usize>;
    //~^ ERROR struct takes 1 lifetime argument but 2 lifetime arguments
    //~| ERROR struct takes 1 generic argument but 2 generic arguments
    //~| HELP remove this lifetime argument
    //~| HELP remove this generic argument
}

mod type_and_type_and_type {
    struct Ty<A, B, C = &'static str>;

    type A = Ty;
    //~^ ERROR missing generics for struct `type_and_type_and_type::Ty`
    //~| HELP add missing

    type B = Ty<usize>;
    //~^ ERROR struct takes at least 2
    //~| HELP add missing

    type C = Ty<usize, String>;

    type D = Ty<usize, String, char>;

    type E = Ty<usize, String, char, f64>;
    //~^ ERROR struct takes at most 3
    //~| HELP remove

    type F = Ty<>;
    //~^ ERROR struct takes at least 2 generic arguments but 0 generic arguments
    //~| HELP add missing
}

// Traits have an implicit `Self` type - these tests ensure we don't accidentally return it
// somewhere in the message
mod r#trait {
    trait NonGeneric {
        //
    }

    trait GenericLifetime<'a> {
        //
    }

    trait GenericType<A> {
        //
    }

    type A = Box<dyn NonGeneric<usize>>;
    //~^ ERROR trait takes 0 generic arguments but 1 generic argument
    //~| HELP remove

    type B = Box<dyn GenericLifetime>;
    //~^ ERROR missing lifetime specifier
    //~| HELP consider introducing
    //~| HELP consider making the bound lifetime-generic

    type C = Box<dyn GenericLifetime<'static, 'static>>;
    //~^ ERROR trait takes 1 lifetime argument but 2 lifetime arguments were supplied
    //~| HELP remove

    type D = Box<dyn GenericType>;
    //~^ ERROR missing generics for trait `GenericType`
    //~| HELP add missing

    type E = Box<dyn GenericType<String, usize>>;
    //~^ ERROR trait takes 1 generic argument but 2 generic arguments
    //~| HELP remove

    type F = Box<dyn GenericLifetime<>>;
    //~^ ERROR missing lifetime specifier
    //~| HELP consider introducing
    //~| HELP consider making the bound lifetime-generic

    type G = Box<dyn GenericType<>>;
    //~^ ERROR trait takes 1 generic argument but 0 generic arguments
    //~| HELP add missing
}

mod associated_item {
    mod non_generic {
        trait NonGenericAT {
            type AssocTy;
        }

        type A = Box<dyn NonGenericAT<usize, AssocTy=()>>;
        //~^ ERROR trait takes 0 generic arguments but 1 generic argument
        //~| HELP remove
    }

    mod lifetime {
        trait GenericLifetimeAT<'a> {
            type AssocTy;
        }

        type A = Box<dyn GenericLifetimeAT<AssocTy=()>>;
        //~^ ERROR missing lifetime specifier
        //~| HELP consider introducing
        //~| HELP consider making the bound lifetime-generic

        type B = Box<dyn GenericLifetimeAT<'static, 'static, AssocTy=()>>;
        //~^ ERROR trait takes 1 lifetime argument but 2 lifetime arguments were supplied
        //~| HELP remove

        type C = Box<dyn GenericLifetimeAT<(), AssocTy=()>>;
        //~^ ERROR missing lifetime specifier
        //~| HELP consider introducing
        //~| HELP consider making the bound lifetime-generic
        //~| ERROR trait takes 0 generic arguments but 1 generic argument
        //~| HELP remove
    }

    mod r#type {
        trait GenericTypeAT<A> {
            type AssocTy;
        }

        type A = Box<dyn GenericTypeAT<AssocTy=()>>;
        //~^ ERROR trait takes 1 generic argument but 0 generic arguments
        //~| HELP add missing

        type B = Box<dyn GenericTypeAT<(), (), AssocTy=()>>;
        //~^ ERROR trait takes 1 generic argument but 2 generic arguments
        //~| HELP remove

        type C = Box<dyn GenericTypeAT<'static, AssocTy=()>>;
        //~^ ERROR trait takes 1 generic argument but 0 generic arguments
        //~| HELP add missing
        //~| ERROR trait takes 0 lifetime arguments but 1 lifetime argument was supplied
        //~| HELP remove
    }

    mod lifetime_and_type {
        trait GenericLifetimeTypeAT<'a, A> {
            type AssocTy;
        }

        type A = Box<dyn GenericLifetimeTypeAT<AssocTy=()>>;
        //~^ ERROR trait takes 1 generic argument but 0 generic arguments
        //~| HELP add missing
        //~| ERROR missing lifetime specifier
        //~| HELP consider introducing
        //~| HELP consider making the bound lifetime-generic

        type B = Box<dyn GenericLifetimeTypeAT<'static, AssocTy=()>>;
        //~^ ERROR trait takes 1 generic argument but 0 generic arguments were supplied
        //~| HELP add missing

        type C = Box<dyn GenericLifetimeTypeAT<'static, 'static, AssocTy=()>>;
        //~^ ERROR trait takes 1 lifetime argument but 2 lifetime arguments were supplied
        //~| HELP remove
        //~| ERROR trait takes 1 generic argument but 0 generic arguments
        //~| HELP add missing

        type D = Box<dyn GenericLifetimeTypeAT<(), AssocTy=()>>;
        //~^ ERROR missing lifetime specifier
        //~| HELP consider introducing
        //~| HELP consider making the bound lifetime-generic

        type E = Box<dyn GenericLifetimeTypeAT<(), (), AssocTy=()>>;
        //~^ ERROR missing lifetime specifier
        //~| HELP consider introducing
        //~| HELP consider making the bound lifetime-generic
        //~| ERROR trait takes 1 generic argument but 2 generic arguments
        //~| HELP remove

        type F = Box<dyn GenericLifetimeTypeAT<'static, 'static, (), AssocTy=()>>;
        //~^ ERROR trait takes 1 lifetime argument but 2 lifetime arguments were supplied
        //~| HELP remove

        type G = Box<dyn GenericLifetimeTypeAT<'static, (), (), AssocTy=()>>;
        //~^ ERROR trait takes 1 generic argument but 2 generic arguments
        //~| HELP remove

        type H = Box<dyn GenericLifetimeTypeAT<'static, 'static, (), (), AssocTy=()>>;
        //~^ ERROR trait takes 1 lifetime argument but 2 lifetime arguments were supplied
        //~| HELP remove
        //~| ERROR trait takes 1 generic argument but 2 generic arguments
        //~| HELP remove
    }

    mod type_and_type {
        trait GenericTypeTypeAT<A, B> {
            type AssocTy;
        }

        type A = Box<dyn GenericTypeTypeAT<AssocTy=()>>;
        //~^ ERROR trait takes 2 generic arguments but 0 generic arguments
        //~| HELP add missing

        type B = Box<dyn GenericTypeTypeAT<(), AssocTy=()>>;
        //~^ ERROR trait takes 2 generic arguments but 1 generic argument
        //~| HELP add missing

        type C = Box<dyn GenericTypeTypeAT<(), (), (), AssocTy=()>>;
        //~^ ERROR trait takes 2 generic arguments but 3 generic arguments
        //~| HELP remove
    }

    mod lifetime_and_lifetime {
        trait GenericLifetimeLifetimeAT<'a, 'b> {
            type AssocTy;
        }

        type A = Box<dyn GenericLifetimeLifetimeAT<AssocTy=()>>;
        //~^ ERROR missing lifetime specifier
        //~| HELP consider introducing
        //~| HELP consider making the bound lifetime-generic

        type B = Box<dyn GenericLifetimeLifetimeAT<'static, AssocTy=()>>;
        //~^ ERROR trait takes 2 lifetime arguments but 1 lifetime argument was supplied
        //~| HELP add missing lifetime argument
    }

    mod lifetime_and_lifetime_and_type {
        trait GenericLifetimeLifetimeTypeAT<'a, 'b, A> {
            type AssocTy;
        }

        type A = Box<dyn GenericLifetimeLifetimeTypeAT<AssocTy=()>>;
        //~^ ERROR missing lifetime specifier
        //~| HELP consider introducing
        //~| HELP consider making the bound lifetime-generic
        //~| ERROR trait takes 1 generic argument but 0 generic arguments
        //~| HELP add missing

        type B = Box<dyn GenericLifetimeLifetimeTypeAT<'static, AssocTy=()>>;
        //~^ ERROR trait takes 2 lifetime arguments but 1 lifetime argument was supplied
        //~| HELP add missing lifetime argument
        //~| ERROR trait takes 1 generic argument but 0 generic arguments
        //~| HELP add missing

        type C = Box<dyn GenericLifetimeLifetimeTypeAT<'static, (), AssocTy=()>>;
        //~^ ERROR trait takes 2 lifetime arguments but 1 lifetime argument was supplied
        //~| HELP add missing lifetime argument
    }
}

mod stdlib {
    mod hash_map {
        use std::collections::HashMap;

        type A = HashMap;
        //~^ ERROR missing generics for struct `HashMap`
        //~| HELP add missing

        type B = HashMap<String>;
        //~^ ERROR struct takes at least
        //~| HELP add missing

        type C = HashMap<'static>;
        //~^ ERROR struct takes 0 lifetime arguments but 1 lifetime argument
        //~| HELP remove these generics
        //~| ERROR struct takes at least 2
        //~| HELP add missing

        type D = HashMap<usize, String, char, f64>;
        //~^ ERROR struct takes at most 3
        //~| HELP remove this

        type E = HashMap<>;
        //~^ ERROR struct takes at least 2 generic arguments but 0 generic arguments
        //~| HELP add missing
    }

    mod result {
        type A = Result;
        //~^ ERROR missing generics for enum `Result`
        //~| HELP add missing

        type B = Result<String>;
        //~^ ERROR enum takes 2 generic arguments but 1 generic argument
        //~| HELP add missing

        type C = Result<'static>;
        //~^ ERROR enum takes 0 lifetime arguments but 1 lifetime argument
        //~| HELP remove these generics
        //~| ERROR enum takes 2 generic arguments but 0 generic arguments
        //~| HELP add missing

        type D = Result<usize, String, char>;
        //~^ ERROR enum takes 2 generic arguments but 3 generic arguments
        //~| HELP remove

        type E = Result<>;
        //~^ ERROR enum takes 2 generic arguments but 0 generic arguments
        //~| HELP add missing
    }
}

fn main() { }
