macro_rules! vec {
    ($($item:expr),*) => 
    {
        {
            let mut v = Vec::new();
            $(
                v.push($item);
            )*
            v
        }
    };
}

