//@ reference: attributes.diagnostic.on_unimplemented.intro
//@ reference: attributes.diagnostic.on_unimplemented.keys
#[diagnostic::on_unimplemented(message = "my message", label = "my label", note = "my note")]
pub trait ProviderLt {}

pub trait ProviderExt {
    fn request<R>(&self) {
        todo!()
    }
}

impl<T: ?Sized + ProviderLt> ProviderExt for T {}

struct B;

fn main() {
    B.request();
    //~^ ERROR my message [E0599]
    //~| NOTE_NONVIRAL my label
    //~| NOTE_NONVIRAL my note
}
