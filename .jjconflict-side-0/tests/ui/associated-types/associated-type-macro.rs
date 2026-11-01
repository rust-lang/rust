fn main() {
    #[cfg(false)]
    <() as module>::mac!(); //~ ERROR macros cannot use qualified paths
}
