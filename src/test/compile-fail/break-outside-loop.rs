// error-pattern:`break` outside of loop
fn main() {
    let pth = break;

    let rs: {t: str} = {t: pth};

}
