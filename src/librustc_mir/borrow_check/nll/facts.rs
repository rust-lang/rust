use crate::borrow_check::location::{LocationIndex, LocationTable};
use crate::dataflow::indexes::BorrowIndex;
use polonius_engine::AllFacts as PoloniusAllFacts;
use polonius_engine::Atom;
use rustc::ty::{RegionVid, TyCtxt};
use rustc_data_structures::indexed_vec::Idx;
use std::error::Error;
use std::fmt::Debug;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

crate type AllFacts = PoloniusAllFacts<RegionVid, BorrowIndex, LocationIndex>;

crate trait AllFactsExt {
    /// Returns `true` if there is a need to gather `AllFacts` given the
    /// current `-Z` flags.
    fn enabled(tcx: TyCtxt<'_>) -> bool;

    fn write_to_dir(
        &self,
        dir: impl AsRef<Path>,
        location_table: &LocationTable,
    ) -> Result<(), Box<dyn Error>>;
}

impl AllFactsExt for AllFacts {
    /// Return
    fn enabled(tcx: TyCtxt<'_>) -> bool {
        tcx.sess.opts.debugging_opts.nll_facts
            || tcx.sess.opts.debugging_opts.polonius
    }

    fn write_to_dir(
        &self,
        dir: impl AsRef<Path>,
        location_table: &LocationTable,
    ) -> Result<(), Box<dyn Error>> {
        let dir: &Path = dir.as_ref();
        fs::create_dir_all(dir)?;
        let wr = FactWriter { location_table, dir };
        macro_rules! write_facts_to_path {
            ($wr:ident . write_facts_to_path($this:ident . [
                $($field:ident,)*
            ])) => {
                $(
                    $wr.write_facts_to_path(
                        &$this.$field,
                        &format!("{}.facts", stringify!($field))
                    )?;
                )*
            }
        }
        write_facts_to_path! {
            wr.write_facts_to_path(self.[
                borrow_region,
                universal_region,
                cfg_edge,
                killed,
                outlives,
                region_live_at,
                invalidates,
            ])
        }
        Ok(())
    }
}

impl Atom for BorrowIndex {
    fn index(self) -> usize {
        Idx::index(self)
    }
}

impl Atom for LocationIndex {
    fn index(self) -> usize {
        Idx::index(self)
    }
}

struct FactWriter<'w> {
    location_table: &'w LocationTable,
    dir: &'w Path,
}

impl<'w> FactWriter<'w> {
    fn write_facts_to_path<T>(
        &self,
        rows: &[T],
        file_name: &str,
    ) -> Result<(), Box<dyn Error>>
    where
        T: FactRow,
    {
        let file = &self.dir.join(file_name);
        let mut file = File::create(file)?;
        for row in rows {
            row.write(&mut file, self.location_table)?;
        }
        Ok(())
    }
}

trait FactRow {
    fn write(
        &self,
        out: &mut File,
        location_table: &LocationTable,
    ) -> Result<(), Box<dyn Error>>;
}

impl FactRow for RegionVid {
    fn write(
        &self,
        out: &mut File,
        location_table: &LocationTable,
    ) -> Result<(), Box<dyn Error>> {
        write_row(out, location_table, &[self])
    }
}

impl<A, B> FactRow for (A, B)
where
    A: FactCell,
    B: FactCell,
{
    fn write(
        &self,
        out: &mut File,
        location_table: &LocationTable,
    ) -> Result<(), Box<dyn Error>> {
        write_row(out, location_table, &[&self.0, &self.1])
    }
}

impl<A, B, C> FactRow for (A, B, C)
where
    A: FactCell,
    B: FactCell,
    C: FactCell,
{
    fn write(
        &self,
        out: &mut File,
        location_table: &LocationTable,
    ) -> Result<(), Box<dyn Error>> {
        write_row(out, location_table, &[&self.0, &self.1, &self.2])
    }
}

impl<A, B, C, D> FactRow for (A, B, C, D)
where
    A: FactCell,
    B: FactCell,
    C: FactCell,
    D: FactCell,
{
    fn write(
        &self,
        out: &mut File,
        location_table: &LocationTable,
    ) -> Result<(), Box<dyn Error>> {
        write_row(out, location_table, &[&self.0, &self.1, &self.2, &self.3])
    }
}

fn write_row(
    out: &mut dyn Write,
    location_table: &LocationTable,
    columns: &[&dyn FactCell],
) -> Result<(), Box<dyn Error>> {
    for (index, c) in columns.iter().enumerate() {
        let tail = if index == columns.len() - 1 {
            "\n"
        } else {
            "\t"
        };
        write!(out, "{:?}{}", c.to_string(location_table), tail)?;
    }
    Ok(())
}

trait FactCell {
    fn to_string(&self, location_table: &LocationTable) -> String;
}

impl<A: Debug> FactCell for A {
    default fn to_string(&self, _location_table: &LocationTable) -> String {
        format!("{:?}", self)
    }
}

impl FactCell for LocationIndex {
    fn to_string(&self, location_table: &LocationTable) -> String {
        format!("{:?}", location_table.to_location(*self))
    }
}
