fn main() {
    let mut ref x = 10; //~  ERROR [E0658]
    x = &11;
    let ref mut y = 12;
    *y = 13;
    let mut ref mut z = 14; //~  ERROR [E0658]
    z = &mut 15;

    #[cfg(false)]
    let mut ref x = 10; //~  ERROR [E0658]
    #[cfg(false)]
    let mut ref mut y = 10; //~  ERROR [E0658]
}
