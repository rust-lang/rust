#[macro_export]
macro_rules! dep_data {
    () => {
        include_str!("dep_data.txt")
    };
}

pub fn dep_data_len() -> usize {
    dep_data!().len()
}

#[macro_export]
macro_rules! dep_bytes {
    () => {
        include_bytes!("dep_bytes.txt")
    };
}

pub fn dep_bytes_len() -> usize {
    dep_bytes!().len()
}
