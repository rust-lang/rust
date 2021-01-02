mod no_generics {
    struct Ty;

    type A = Ty;

    type B = Ty<'static>;
    //~^ ERROR this struct takes 0 lifetime arguments but 1 lifetime argument was supplied
    //~| HELP remove these generics

    type C = Ty<'static, usize>;
    //~^ ERROR this struct takes 0 lifetime arguments but 1 lifetime argument was supplied
    //~| ERROR this struct takes 0 type arguments but 1 type argument was supplied
    //~| HELP remove this lifetime argument
    //~| HELP remove this type argument

    type D = Ty<'static, usize, { 0 }>;
    //~^ ERROR this struct takes 0 lifetime arguments but 1 lifetime argument was supplied
    //~| ERROR this struct takes 0 generic arguments but 2 generic arguments were supplied
    //~| HELP remove this lifetime argument
    //~| HELP remove these generic arguments
}

mod type_and_type {
    struct Ty<A, B>;

    type A = Ty;
    //~^ ERROR missing generics for struct `type_and_type::Ty`
    //~| HELP use angle brackets

    type B = Ty<usize>;
    //~^ ERROR this struct takes 2 type arguments but only 1 type argument was supplied
    //~| HELP add missing type argument

    type C = Ty<usize, String>;

    type D = Ty<usize, String, char>;
    //~^ ERROR this struct takes 2 type arguments but 3 type arguments were supplied
    //~| HELP remove this type argument
}

mod lifetime_and_type {
    struct Ty<'a, T>;

    type A = Ty;
    //~^ ERROR missing generics for struct `lifetime_and_type::Ty`
    //~| ERROR missing lifetime specifier
    //~| HELP consider introducing
    //~| HELP use angle brackets

    type B = Ty<'static>;
    //~^ ERROR this struct takes 1 type argument but 0 type arguments were supplied
    //~| HELP add missing type argument

    type C = Ty<usize>;
    //~^ ERROR missing lifetime specifier
    //~| HELP consider introducing

    type D = Ty<'static, usize>;
}

mod type_and_type_and_type {
    struct Ty<A, B, C = &'static str>;

    type A = Ty;
    //~^ ERROR missing generics for struct `type_and_type_and_type::Ty`
    //~| HELP use angle brackets

    type B = Ty<usize>;
    //~^ ERROR this struct takes at least 2 type arguments but only 1 type argument was supplied
    //~| HELP add missing type argument

    type C = Ty<usize, String>;

    type D = Ty<usize, String, char>;

    type E = Ty<usize, String, char, f64>;
    //~^ ERROR this struct takes at most 3 type arguments but 4 type arguments were supplied
    //~| HELP remove
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
    //~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied
    //~| HELP remove

    type B = Box<dyn GenericLifetime>;
    //~^ ERROR missing lifetime specifier
    //~| HELP consider introducing

    type C = Box<dyn GenericLifetime<'static, 'static>>;
    //~^ ERROR this trait takes 1 lifetime argument but 2 lifetime arguments were supplied
    //~| HELP remove

    type D = Box<dyn GenericType>;
    //~^ ERROR missing generics for trait `GenericType`
    //~| HELP use angle brackets

    type E = Box<dyn GenericType<String, usize>>;
    //~^ ERROR this trait takes 1 type argument but 2 type arguments were supplied
    //~| HELP remove
}

mod stdlib {
    mod hash_map {
        use std::collections::HashMap;

        type A = HashMap;
        //~^ ERROR missing generics for struct `HashMap`
        //~| HELP use angle brackets

        type B = HashMap<String>;
        //~^ ERROR this struct takes at least 2 type arguments but only 1 type argument was supplied
        //~| HELP add missing type argument

        type C = HashMap<'static>;
        //~^ ERROR this struct takes 0 lifetime arguments but 1 lifetime argument was supplied
        //~| HELP remove these generics
        //~| ERROR this struct takes at least 2 type arguments but 0 type arguments were supplied
        //~| HELP add missing type arguments

        type D = HashMap<usize, String, char, f64>;
        //~^ ERROR this struct takes at most 3 type arguments but 4 type arguments were supplied
        //~| HELP remove this type argument
    }

    mod result {
        type A = Result;
        //~^ ERROR missing generics for enum `std::result::Result`
        //~| HELP use angle brackets

        type B = Result<String>;
        //~^ ERROR this enum takes 2 type arguments but only 1 type argument was supplied
        //~| HELP add missing type argument

        type C = Result<'static>;
        //~^ ERROR this enum takes 0 lifetime arguments but 1 lifetime argument was supplied
        //~| HELP remove these generics
        //~| ERROR this enum takes 2 type arguments but 0 type arguments were supplied
        //~| HELP add missing type arguments

        type D = Result<usize, String, char>;
        //~^ ERROR this enum takes 2 type arguments but 3 type arguments were supplied
        //~| HELP remove
    }
}

fn main() { }
