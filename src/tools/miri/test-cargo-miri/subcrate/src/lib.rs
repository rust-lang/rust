#[cfg(doctest)]
compile_error!("rustdoc should not touch me");

#[cfg(test)]
compile_error!("Miri should not touch me");
