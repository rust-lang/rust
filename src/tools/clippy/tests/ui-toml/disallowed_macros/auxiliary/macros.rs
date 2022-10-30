#[macro_export]
macro_rules! expr {
    () => {
        1
    };
}

#[macro_export]
macro_rules! stmt {
    () => {
        let _x = ();
    };
}

#[macro_export]
macro_rules! ty {
    () => { &'static str };
}

#[macro_export]
macro_rules! pat {
    () => {
        _
    };
}

#[macro_export]
macro_rules! item {
    () => {
        const ITEM: usize = 1;
    };
}
