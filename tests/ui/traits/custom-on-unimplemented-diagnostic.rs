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
    //~^ my message [E0599]
    //~| my label
    //~| my note
}
