#[cfg(doctest)]
use num_cpus as _;

#[cfg(test)]
compile_error!("Miri should not touch me");
