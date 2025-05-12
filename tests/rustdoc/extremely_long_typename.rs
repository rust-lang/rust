// ignore-tidy-linelength
// Make sure that, if an extremely long type name is named,
// the item table has it line wrapped.
// There should be some reasonably-placed `<wbr>` tags in the snapshot file.

//@ snapshot extremely_long_typename "extremely_long_typename/index.html" '//dl[@class="item-table"]/dt'
pub struct CreateSubscriptionPaymentSettingsPaymentMethodOptionsCustomerBalanceBankTransferEuBankTransfer;
