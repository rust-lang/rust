//@ edition: 2021

struct NotClonableArg;
#[derive(Default)]
struct NotClonableReturnType;

// Verify that the only components that we care about are the upvars, not the signature.
fn we_are_okay_with_not_clonable_signature() {
    let x = async |x: NotClonableArg| -> NotClonableReturnType { Default::default() };
    x.clone(); // Okay
}

#[derive(Debug)]
struct NotClonableUpvar;

fn we_only_care_about_clonable_upvars() {
    let x = NotClonableUpvar;
    // Notably, this is clone because we capture `&x`.
    let yes_clone = async || {
        println!("{x:?}");
    };
    yes_clone.clone(); // Okay

    let z = NotClonableUpvar;
    // However, this is not because the closure captures `z` by move.
    // (Even though the future that is lent out captures `z by ref!)
    let not_clone = async move || {
        println!("{z:?}");
    };
    not_clone.clone();
    //~^ ERROR the trait bound `NotClonableUpvar: Clone` is not satisfied
}

fn main() {}
