fn main() {
    #[cfg(FALSE)]
    <() as module>::mac!(); //~ ERROR macros cannot use qualified paths
}
