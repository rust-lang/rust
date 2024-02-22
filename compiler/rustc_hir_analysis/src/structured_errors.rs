mod missing_cast_for_variadic_arg;
mod sized_unsized_cast;
mod wrong_number_of_generic_args;

pub use self::{
    missing_cast_for_variadic_arg::*, sized_unsized_cast::*, wrong_number_of_generic_args::*,
};

use rustc_errors::{Diag, ErrCode};
use rustc_session::Session;

pub trait StructuredDiagnostic<'tcx> {
    fn session(&self) -> &Session;

    fn code(&self) -> ErrCode;

    fn diagnostic(&self) -> Diag<'tcx> {
        let err = self.diagnostic_common();

        if self.session().teach(self.code()) {
            self.diagnostic_extended(err)
        } else {
            self.diagnostic_regular(err)
        }
    }

    fn diagnostic_common(&self) -> Diag<'tcx>;

    fn diagnostic_regular(&self, err: Diag<'tcx>) -> Diag<'tcx> {
        err
    }

    fn diagnostic_extended(&self, err: Diag<'tcx>) -> Diag<'tcx> {
        err
    }
}
