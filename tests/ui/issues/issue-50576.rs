fn main() {
    |bool: [u8; break 'L]| 0;
    //~^ ERROR [E0426]
    //~| ERROR [E0268]
    Vec::<[u8; break]>::new(); //~ ERROR [E0268]
}
