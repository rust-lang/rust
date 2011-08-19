// error-pattern:Break outside a loop
fn main() {
    let pth = break;

    let rs: {t: str} = {t: pth};

}
