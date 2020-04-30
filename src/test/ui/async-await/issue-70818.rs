// edition 2018

fn d<T: Sized>(t: T) -> impl std::future::Future<Output = T> + Send { //~ Error `T` cannot be sent between threads safely
    async { t }
}
