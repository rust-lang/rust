// edition 2018

fn foo<T: Sized>(ty: T) -> impl std::future::Future<Output = T> + Send { //~ Error `T` cannot be sent between threads safely
    async { ty }
}

fn main() {}
