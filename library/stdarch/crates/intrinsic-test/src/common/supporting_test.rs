/// Architectures must support this trait
/// to be successfully tested.
pub trait SupportedArchitectureTest {
    fn write_c_file(filename: &str);

    fn write_rust_file(filename: &str);

    fn build_c_file(filename: &str);

    fn build_rust_file(filename: &str);

    fn read_intrinsic_source_file(filename: &str);
}
