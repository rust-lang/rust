pub struct ProjectileCreated;
pub struct NotificationChannel<E>(std::marker::PhantomData<E>);

// The inner `macro_rules!` is what later reports a span from this crate while the
// diagnostic is being rendered against the downstream crate's source.
macro_rules! define_trigger_system {
    ($(( $field:ident, $ty:ident, $channel:ident )),* $(,)?) => {
        #[macro_export]
        macro_rules! all_trigger_fields {
            ($submacro:ident) => { $submacro!($( ( $field, $ty, $channel ) ),*) }
        }
    };
}

define_trigger_system!((projectile_created, ProjectileCreated, NotificationChannel),);
