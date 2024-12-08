fn main() {
    let 5 = 6;
    //~^ error refutable pattern in local binding [E0005]

    let x @ 5 = 6;
    //~^ error refutable pattern in local binding [E0005]
}
