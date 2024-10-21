//@ known-bug: #131538
#![feature(generic_associated_types_extended)]
#![feature(trivial_bounds)]

trait HealthCheck {
    async fn check<const N: usize>();
}

fn do_health_check_par()
where
    HealthCheck: HealthCheck,
{
}
