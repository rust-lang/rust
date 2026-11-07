// issue #92069


fn main() {
    let nums = vec![5, 4, 3, 2, 1];
    let [x, y] = nums else { //~ ERROR expected an array or slice
        return;
    };
}
