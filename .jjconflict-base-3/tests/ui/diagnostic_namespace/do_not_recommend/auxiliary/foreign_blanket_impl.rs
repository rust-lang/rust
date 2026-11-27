pub struct ForeignType<T>(pub T);

pub trait DoNotMentionThis {}

#[diagnostic::do_not_recommend]
impl<T: DoNotMentionThis> Clone for ForeignType<T> {
    fn clone(&self) -> Self {
        todo!()
    }
}
