pub struct Something;

type Validator<'a> = dyn 'a + Send + Sync + Fn(&'a Something) -> Result<(), ()>;

pub static ALL_VALIDATORS: &[(&'static str, &'static Validator)] =
    &[("validate that credits and debits balance", &validate_something)];

fn or<'a>(first: &'static Validator<'a>, second: &'static Validator<'a>) -> Validator<'a> {
    //~^ ERROR return type cannot be a trait object without pointer indirection
    return Box::new(move |something: &'_ Something| -> Result<(), ()> {
        first(something).or_else(|_| second(something))
    });
}

fn validate_something(_: &Something) -> Result<(), ()> {
    Ok(())
}

fn main() {}
