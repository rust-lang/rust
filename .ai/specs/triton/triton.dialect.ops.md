# Triton Dialect Operations (from TableGen)

Source scanned: `src/triton/include/triton/Dialect/Triton` (`*.td`)

## Triton ops (`tt.*`)

- `tt.addptr`
- `tt.advance`
- `tt.assert`
- `tt.atomic_cas`
- `tt.atomic_rmw`
- `tt.bitcast`
- `tt.broadcast`
- `tt.call`
- `tt.cat`
- `tt.clampf`
- `tt.descriptor_gather`
- `tt.descriptor_load`
- `tt.descriptor_reduce`
- `tt.descriptor_scatter`
- `tt.descriptor_store`
- `tt.dot`
- `tt.dot_scaled`
- `tt.elementwise_inline_asm`
- `tt.expand_dims`
- `tt.extern_elementwise`
- `tt.fp_to_fp`
- `tt.func`
- `tt.gather`
- `tt.get_num_programs`
- `tt.get_program_id`
- `tt.histogram`
- `tt.int_to_ptr`
- `tt.join`
- `tt.load`
- `tt.make_range`
- `tt.make_tensor_descriptor`
- `tt.make_tensor_ptr`
- `tt.map_elementwise`
- `tt.map_elementwise.return`
- `tt.mulhiui`
- `tt.precise_divf`
- `tt.precise_sqrt`
- `tt.print`
- `tt.ptr_to_int`
- `tt.reduce`
- `tt.reduce.return`
- `tt.reshape`
- `tt.return`
- `tt.scan`
- `tt.scan.return`
- `tt.splat`
- `tt.split`
- `tt.store`
- `tt.trans`
- `tt.unsplat`

## Other dialect ops referenced in this folder

- `arith.bitcast`
- `scf.for`
- `scf.while`
