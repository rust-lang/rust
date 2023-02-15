use super::ConstraintCategory;

impl<'tcx> ConstraintCategory<'tcx> {
    pub fn cmp_discr(&self) -> u8 {
        use ConstraintCategory::*;
        match self {
            Return(_) => 0,
            Yield => 1,
            UseAsConst => 2,
            UseAsStatic => 3,
            TypeAnnotation => 4,
            Cast => 5,
            ClosureBounds => 6,
            CallArgument(_) => 7,
            CopyBound => 8,
            SizedBound => 9,
            Assignment => 10,
            Usage => 11,
            OpaqueType => 12,
            ClosureUpvar(_) => 13,
            Predicate(_) => 14,
            Boring => 15,
            BoringNoLocation => 16,
            Internal => 17,
        }
    }
}
