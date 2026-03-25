/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use melior::Context;
use melior::ir::Location;
use melior::ir::r#type::IntegerType;

use crate::errors::Error;
use crate::triton::attr_i32;
use crate::triton::tt::GetProgramIdOperation;

#[derive(Debug, Clone, Copy)]
pub enum ProgramAxis {
    Axis0 = 0,
    Axis1 = 1,
    Axis2 = 2,
}

impl From<i32> for ProgramAxis {
    fn from(value: i32) -> Self {
        match value {
            0 => ProgramAxis::Axis0,
            1 => ProgramAxis::Axis1,
            2 => ProgramAxis::Axis2,
            _ => panic!("Invalid program axis: {}", value),
        }
    }
}

pub fn create_get_program_id<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    axis: ProgramAxis,
) -> Result<GetProgramIdOperation<'ctx>, Error> {
    let axis_attr = attr_i32(context, axis as i32).into();
    let result = IntegerType::new(context, 32).into();
    Ok(GetProgramIdOperation::builder(context, location).axis(axis_attr).result(result).build())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tt_program_id() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let axis = ProgramAxis::Axis1;

        let program_id_op = create_get_program_id(&context, location, axis);

        let output = program_id_op.unwrap().as_operation().to_string();
        let expected = "tt.get_program_id\"() {axis = 1 : i32} : () -> i32\n";
        assert_eq!(expected, output);
    }
}
