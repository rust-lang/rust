set -x
cargo install --force --path .

echo "Running integration test for crate ${INTEGRATION}"

git clone --depth=1 https://github.com/${INTEGRATION}.git checkout
cd checkout

function check() {
  RUST_BACKTRACE=full cargo clippy --all &> clippy_output
  cat clippy_output
  ! cat clippy_output | grep -q "internal compiler error"
  if [[ $? != 0 ]]; then
    return 1
  fi
}

case ${INTEGRATION} in
  rust-lang/cargo)
    check
    ;;
  *)
    check
    ;;
esac
