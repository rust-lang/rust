// rustfmt-fn_call_indent: Block
// rustfmt-max_width: 80
// rustfmt-tab_spaces: 2

// #1427
fn main() {
  exceptaions::config(move || {
    (
      NmiConfig {},
      HardFaultConfig {},
      SysTickConfig { gpio_sbsrr },
    )
  });
}
