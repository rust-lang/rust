//@rustc-env: CLIPPY_PETS_PRINT=1
//@rustc-env: CLIPPY_PRINT_MIR=1

struct Dropper {
    duck: u32,
}

impl Drop for Dropper {
    fn drop(&mut self) {}
}

struct Animal {
    legs: u32,
    heads: u32,
}

// Check arguments are correctly detected
#[warn(clippy::borrow_pats)]
fn take_one(_animal: Animal) {}

#[warn(clippy::borrow_pats)]
fn take_two(_animal_1: Animal, _animal_2: Animal) {}

fn take_pair((_animal_1, _animal_2): (Animal, Animal)) {}

#[warn(clippy::borrow_pats)]
fn pat_return_owned_arg(animal: Animal) -> Animal {
    animal
}

#[warn(clippy::borrow_pats)]
fn pat_maybe_return_owned_arg_1(a: String) -> String {
    if !a.is_empty() {
        return a;
    }

    "hey".to_string()
}

#[warn(clippy::borrow_pats)]
fn pat_maybe_return_owned_arg_1_test(a: u32) -> u32 {
    if !a.is_power_of_two() {
        return a;
    }

    19
}

#[warn(clippy::borrow_pats)]
/// FIXME: The argument return is not yet detected both in `a`
fn pat_maybe_return_owned_arg_2(a: String) -> String {
    let ret;
    if !a.is_empty() {
        ret = a;
    } else {
        ret = "hey".to_string();
        println!("{a:#?}");
    }
    ret
}

#[warn(clippy::borrow_pats)]
fn pat_maybe_return_owned_arg_3(a: String) -> String {
    let ret = if !a.is_empty() { a } else { "hey".to_string() };
    ret
}

#[warn(clippy::borrow_pats)]
fn pub_dynamic_drop_1(animal: String, cond: bool) {
    if cond {
        // Move out of function
        std::mem::drop(animal);
    }

    magic()
}

#[warn(clippy::borrow_pats)]
fn conditional_overwrite(mut animal: String, cond: bool) {
    if cond {
        animal = "Ducks".to_string();
    }

    magic()
}

fn magic() {}

#[derive(Default)]
struct Example {
    owned_1: String,
    owned_2: String,
    copy_1: u32,
    copy_2: u32,
}

#[warn(clippy::borrow_pats)]
fn test_ctors() {
    let s1 = String::new();
    let _slice = (s1, 2);

    let s1 = String::new();
    let _array = [s1];

    let s1 = String::new();
    let _thing = Example {
        owned_1: s1,
        ..Example::default()
    };
}

#[warn(clippy::borrow_pats)]
fn main() {
    let dropper = Dropper { duck: 17 };
}
