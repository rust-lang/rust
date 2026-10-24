// Removing `self::self` breaks idempotence, leaving it causes a compilation error
// which the user should be made aware of #6558
use self;
use self::self;
use self::self::self;
