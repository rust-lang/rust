fn foo() {
    let a = [(); const {
        let x = 1;
        x
    }];
}

fn foo() {
    let x = [(); const { 1 }];
}
