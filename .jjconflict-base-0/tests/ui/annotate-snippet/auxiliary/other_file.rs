pub struct WithPrivateMethod;

impl WithPrivateMethod {
    /// Private to get an error involving two files
    fn private_method(&self) {}
}
